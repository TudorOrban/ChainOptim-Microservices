package org.chainoptimsupply.core.supplierorder.service;

import org.chainoptimsupply.core.supplier.dto.SupplierDTOMapper;
import org.chainoptimsupply.core.supplierorder.model.SupplierOrder;
import org.chainoptimsupply.exception.PlanLimitReachedException;
import org.chainoptimsupply.exception.ResourceNotFoundException;
import org.chainoptimsupply.exception.ValidationException;
import org.chainoptimsupply.internal.component.repository.ComponentRepository;
import org.chainoptimsupply.internal.subscriptionplan.model.Component;
import org.chainoptimsupply.internal.subscriptionplan.service.SubscriptionPlanLimiterService;
import org.chainoptimsupply.shared.enums.Feature;
import org.chainoptimsupply.shared.kafka.KafkaEvent;
import org.chainoptimsupply.shared.PaginatedResults;
import org.chainoptimsupply.shared.enums.SearchMode;
import org.chainoptimsupply.shared.sanitization.EntitySanitizerService;
import org.chainoptimsupply.shared.search.SearchParams;
import org.chainoptimsupply.core.supplier.dto.CreateSupplierOrderDTO;
import org.chainoptimsupply.core.supplier.dto.UpdateSupplierOrderDTO;
import org.chainoptimsupply.core.supplierorder.model.SupplierOrderEvent;
import org.chainoptimsupply.core.supplierorder.repository.SupplierOrderRepository;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.util.*;

@Service
public class SupplierOrderServiceImpl implements SupplierOrderService {

    private final SupplierOrderRepository supplierOrderRepository;
    private final KafkaSupplierOrderService kafkaSupplierOrderService;
    private final ComponentRepository componentRepository;
    private final SubscriptionPlanLimiterService planLimiterService;
    private final EntitySanitizerService entitySanitizerService;

    @Autowired
    public SupplierOrderServiceImpl(
            SupplierOrderRepository supplierOrderRepository,
            KafkaSupplierOrderService kafkaSupplierOrderService,
            ComponentRepository componentRepository,
            SubscriptionPlanLimiterService planLimiterService,
            EntitySanitizerService entitySanitizerService
    ) {
        this.supplierOrderRepository = supplierOrderRepository;
        this.kafkaSupplierOrderService = kafkaSupplierOrderService;
        this.componentRepository = componentRepository;
        this.planLimiterService = planLimiterService;
        this.entitySanitizerService = entitySanitizerService;
    }

    public List<SupplierOrder> getSupplierOrdersByOrganizationId(Integer organizationId) {
        return supplierOrderRepository.findByOrganizationId(organizationId);
    }

    public List<SupplierOrder> getSupplierOrdersBySupplierId(Integer supplierId) {
        return supplierOrderRepository.findBySupplierId(supplierId);
    }

    public PaginatedResults<SupplierOrder> getSupplierOrdersAdvanced(SearchMode searchMode, Integer entityId, SearchParams searchParams) {
        // Attempt to parse filters JSON
        Map<String, String> filters;
        if (!searchParams.getFiltersJson().isEmpty()) {
            try {
                filters = new ObjectMapper().readValue(searchParams.getFiltersJson(), new TypeReference<Map<String, String>>(){});
                searchParams.setFilters(filters);
            } catch (JsonProcessingException e) {
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid filters format");
            }
        }

        return supplierOrderRepository.findBySupplierIdAdvanced(searchMode, entityId, searchParams);
    }

    public Integer getOrganizationIdById(Long supplierOrderId) {
        return supplierOrderRepository.findOrganizationIdById(supplierOrderId)
                .orElseThrow(() -> new ResourceNotFoundException("Supplier Order with ID: " + supplierOrderId + " not found."));
    }

    public long countByOrganizationId(Integer organizationId) {
        return supplierOrderRepository.countByOrganizationId(organizationId);
    }

    // Create
    public SupplierOrder createSupplierOrder(CreateSupplierOrderDTO orderDTO) {
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(orderDTO.getOrganizationId(), Feature.SUPPLIER_ORDER, 1)) {
            throw new PlanLimitReachedException("You have reached the limit of allowed Supplier Orders for the current Subscription Plan.");
        }

        // Sanitize input and map to entity
        CreateSupplierOrderDTO sanitizedOrderDTO = entitySanitizerService.sanitizeCreateSupplierOrderDTO(orderDTO);
        SupplierOrder supplierOrder = SupplierDTOMapper.mapCreateDtoToSupplierOrder(sanitizedOrderDTO);
        Component component = componentRepository.findById(sanitizedOrderDTO.getComponentId())
                .orElseThrow(() -> new ResourceNotFoundException("Component with ID: " + sanitizedOrderDTO.getComponentId() + " not found."));
        supplierOrder.setComponentId(component.getId());

        SupplierOrder savedOrder = supplierOrderRepository.save(supplierOrder);

        // Publish order to Kafka broker
        kafkaSupplierOrderService.sendSupplierOrderEvent(
                new SupplierOrderEvent(savedOrder, null, KafkaEvent.EventType.CREATE, savedOrder.getSupplierId(), Feature.SUPPLIER, "Test"));

        return savedOrder;
    }

    @Transactional
    public List<SupplierOrder> createSupplierOrdersInBulk(List<CreateSupplierOrderDTO> orderDTOs) {
        // Ensure same organizationId
        if (orderDTOs.stream().map(CreateSupplierOrderDTO::getOrganizationId).distinct().count() > 1) {
            throw new ValidationException("All orders must belong to the same organization.");
        }
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(orderDTOs.getFirst().getOrganizationId(), Feature.SUPPLIER_ORDER, orderDTOs.size())) {
            throw new PlanLimitReachedException("You have reached the limit of allowed Supplier Orders for the current Subscription Plan.");
        }

        // Sanitize and map to entity
        List<SupplierOrder> orders = orderDTOs.stream()
                .map(orderDTO -> {
                    CreateSupplierOrderDTO sanitizedOrderDTO = entitySanitizerService.sanitizeCreateSupplierOrderDTO(orderDTO);
                    SupplierOrder supplierOrder = SupplierDTOMapper.mapCreateDtoToSupplierOrder(sanitizedOrderDTO);
                    Component component = componentRepository.findById(sanitizedOrderDTO.getComponentId())
                            .orElseThrow(() -> new ResourceNotFoundException("Component with ID: " + sanitizedOrderDTO.getComponentId() + " not found."));
                    supplierOrder.setComponentId(component.getId());
                    return supplierOrder;
                })
                .toList();

        List<SupplierOrder> savedOrders = supplierOrderRepository.saveAll(orders);

        // Publish order events to Kafka broker
        List<SupplierOrderEvent> orderEvents = new ArrayList<>();
        orders.stream()
                .map(order -> new SupplierOrderEvent(order, null, KafkaEvent.EventType.CREATE, order.getSupplierId(), Feature.SUPPLIER, "Test"))
                .forEach(orderEvents::add);

        kafkaSupplierOrderService.sendSupplierOrderEventsInBulk(orderEvents);

        return savedOrders;
    }

    @Transactional
    public List<SupplierOrder> updateSuppliersOrdersInBulk(List<UpdateSupplierOrderDTO> orderDTOs) {
        List<SupplierOrder> orders = supplierOrderRepository.findByIds(orderDTOs.stream().map(UpdateSupplierOrderDTO::getId).toList())
                .orElseThrow(() -> new ResourceNotFoundException("Supplier Orders not found."));

        // Save old orders for event publishing
        Map<Integer, SupplierOrder> oldOrders = new HashMap<>();
        for (SupplierOrder order: orders) {
            oldOrders.put(order.getId(), order.deepCopy());
        }

        // Update orders
        List<SupplierOrder> updatedOrders = orders.stream()
                .map(order -> {
                    UpdateSupplierOrderDTO orderDTO = orderDTOs.stream()
                            .filter(dto -> dto.getId().equals(order.getId()))
                            .findFirst()
                            .orElseThrow(() -> new ResourceNotFoundException("Supplier Order with ID: " + order.getId() + " not found."));
                    UpdateSupplierOrderDTO sanitizedOrderDTO = entitySanitizerService.sanitizeUpdateSupplierOrderDTO(orderDTO);
                    SupplierDTOMapper.setUpdateSupplierOrderDTOToUpdateOrder(order, sanitizedOrderDTO);
                    Component component = componentRepository.findById(sanitizedOrderDTO.getComponentId())
                            .orElseThrow(() -> new ResourceNotFoundException("Component with ID: " + sanitizedOrderDTO.getComponentId() + " not found."));
                    order.setComponentId(component.getId());
                    return order;
                }).toList();

        // Save
        List<SupplierOrder> savedOrders = supplierOrderRepository.saveAll(updatedOrders);

        // Publish order events to Kafka broker
        List<SupplierOrderEvent> orderEvents = new ArrayList<>();
        savedOrders.stream()
                .map(order -> {
                    SupplierOrder oldOrder = oldOrders.get(order.getId());
                    return new SupplierOrderEvent(order, oldOrder, KafkaEvent.EventType.UPDATE, order.getSupplierId(), Feature.SUPPLIER, "Test");
                })
                .forEach(orderEvents::add);

        kafkaSupplierOrderService.sendSupplierOrderEventsInBulk(orderEvents);

        return savedOrders;
    }

    @Transactional
    public List<Integer> deleteSupplierOrdersInBulk(List<Integer> orderIds) {
        List<SupplierOrder> orders = supplierOrderRepository.findAllById(orderIds);
        // Ensure same organizationId
        if (orders.stream().map(SupplierOrder::getOrganizationId).distinct().count() > 1) {
            throw new ValidationException("All orders must belong to the same organization.");
        }

        supplierOrderRepository.deleteAll(orders);

        // Publish order events to Kafka broker
        List<SupplierOrderEvent> orderEvents = new ArrayList<>();
        orders.stream()
                .map(order -> new SupplierOrderEvent(null, order, KafkaEvent.EventType.CREATE, order.getSupplierId(), Feature.SUPPLIER, "Test"))
                .forEach(orderEvents::add);

        kafkaSupplierOrderService.sendSupplierOrderEventsInBulk(orderEvents);

        return orderIds;
    }

}
