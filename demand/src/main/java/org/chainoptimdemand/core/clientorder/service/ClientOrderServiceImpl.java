package org.chainoptimdemand.core.clientorder.service;

import org.chainoptimdemand.core.client.dto.ClientDTOMapper;
import org.chainoptimdemand.core.clientorder.model.ClientOrder;
import org.chainoptimdemand.exception.PlanLimitReachedException;
import org.chainoptimdemand.exception.ResourceNotFoundException;
import org.chainoptimdemand.exception.ValidationException;
import org.chainoptimdemand.internal.in.goods.repository.ComponentRepository;
import org.chainoptimdemand.internal.in.goods.model.Component;
import org.chainoptimdemand.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimdemand.shared.enums.Feature;
import org.chainoptimdemand.shared.kafka.KafkaEvent;
import org.chainoptimdemand.shared.PaginatedResults;
import org.chainoptimdemand.shared.enums.SearchMode;
import org.chainoptimdemand.shared.sanitization.EntitySanitizerService;
import org.chainoptimdemand.shared.search.SearchParams;
import org.chainoptimdemand.core.client.dto.CreateClientOrderDTO;
import org.chainoptimdemand.core.client.dto.UpdateClientOrderDTO;
import org.chainoptimdemand.core.clientorder.model.ClientOrderEvent;
import org.chainoptimdemand.core.clientorder.repository.ClientOrderRepository;

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
public class ClientOrderServiceImpl implements ClientOrderService {

    private final ClientOrderRepository clientOrderRepository;
    private final KafkaClientOrderService kafkaClientOrderService;
    private final ComponentRepository componentRepository;
    private final SubscriptionPlanLimiterService planLimiterService;
    private final EntitySanitizerService entitySanitizerService;

    @Autowired
    public ClientOrderServiceImpl(
            ClientOrderRepository clientOrderRepository,
            KafkaClientOrderService kafkaClientOrderService,
            ComponentRepository componentRepository,
            SubscriptionPlanLimiterService planLimiterService,
            EntitySanitizerService entitySanitizerService
    ) {
        this.clientOrderRepository = clientOrderRepository;
        this.kafkaClientOrderService = kafkaClientOrderService;
        this.componentRepository = componentRepository;
        this.planLimiterService = planLimiterService;
        this.entitySanitizerService = entitySanitizerService;
    }

    public List<ClientOrder> getClientOrdersByOrganizationId(Integer organizationId) {
        return clientOrderRepository.findByOrganizationId(organizationId);
    }

    public List<ClientOrder> getClientOrdersByClientId(Integer clientId) {
        return clientOrderRepository.findByClientId(clientId);
    }

    public PaginatedResults<ClientOrder> getClientOrdersAdvanced(SearchMode searchMode, Integer entityId, SearchParams searchParams) {
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

        return clientOrderRepository.findByClientIdAdvanced(searchMode, entityId, searchParams);
    }

    public Integer getOrganizationIdById(Long clientOrderId) {
        return clientOrderRepository.findOrganizationIdById(clientOrderId)
                .orElseThrow(() -> new ResourceNotFoundException("Client Order with ID: " + clientOrderId + " not found."));
    }

    public long countByOrganizationId(Integer organizationId) {
        return clientOrderRepository.countByOrganizationId(organizationId);
    }

    // Create
    public ClientOrder createClientOrder(CreateClientOrderDTO orderDTO) {
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(orderDTO.getOrganizationId(), Feature.SUPPLIER_ORDER, 1)) {
            throw new PlanLimitReachedException("You have reached the limit of allowed Client Orders for the current Subscription Plan.");
        }

        // Sanitize input and map to entity
        CreateClientOrderDTO sanitizedOrderDTO = entitySanitizerService.sanitizeCreateClientOrderDTO(orderDTO);
        ClientOrder clientOrder = ClientDTOMapper.mapCreateDtoToClientOrder(sanitizedOrderDTO);
        Component component = componentRepository.findById(sanitizedOrderDTO.getComponentId())
                .orElseThrow(() -> new ResourceNotFoundException("Component with ID: " + sanitizedOrderDTO.getComponentId() + " not found."));
        clientOrder.setComponentId(component.getId());

        ClientOrder savedOrder = clientOrderRepository.save(clientOrder);

        // Publish order to Kafka broker
        kafkaClientOrderService.sendClientOrderEvent(
                new ClientOrderEvent(savedOrder, null, KafkaEvent.EventType.CREATE, savedOrder.getClientId(), Feature.SUPPLIER, "Test"));

        return savedOrder;
    }

    @Transactional
    public List<ClientOrder> createClientOrdersInBulk(List<CreateClientOrderDTO> orderDTOs) {
        // Ensure same organizationId
        if (orderDTOs.stream().map(CreateClientOrderDTO::getOrganizationId).distinct().count() > 1) {
            throw new ValidationException("All orders must belong to the same organization.");
        }
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(orderDTOs.getFirst().getOrganizationId(), Feature.SUPPLIER_ORDER, orderDTOs.size())) {
            throw new PlanLimitReachedException("You have reached the limit of allowed Client Orders for the current Subscription Plan.");
        }

        // Sanitize and map to entity
        List<ClientOrder> orders = orderDTOs.stream()
                .map(orderDTO -> {
                    CreateClientOrderDTO sanitizedOrderDTO = entitySanitizerService.sanitizeCreateClientOrderDTO(orderDTO);
                    ClientOrder clientOrder = ClientDTOMapper.mapCreateDtoToClientOrder(sanitizedOrderDTO);
                    Component component = componentRepository.findById(sanitizedOrderDTO.getComponentId())
                            .orElseThrow(() -> new ResourceNotFoundException("Component with ID: " + sanitizedOrderDTO.getComponentId() + " not found."));
                    clientOrder.setComponentId(component.getId());
                    return clientOrder;
                })
                .toList();

        List<ClientOrder> savedOrders = clientOrderRepository.saveAll(orders);

        // Publish order events to Kafka broker
        List<ClientOrderEvent> orderEvents = new ArrayList<>();
        orders.stream()
                .map(order -> new ClientOrderEvent(order, null, KafkaEvent.EventType.CREATE, order.getClientId(), Feature.SUPPLIER, "Test"))
                .forEach(orderEvents::add);

        kafkaClientOrderService.sendClientOrderEventsInBulk(orderEvents);

        return savedOrders;
    }

    @Transactional
    public List<ClientOrder> updateClientsOrdersInBulk(List<UpdateClientOrderDTO> orderDTOs) {
        List<ClientOrder> orders = clientOrderRepository.findByIds(orderDTOs.stream().map(UpdateClientOrderDTO::getId).toList())
                .orElseThrow(() -> new ResourceNotFoundException("Client Orders not found."));

        // Save old orders for event publishing
        Map<Integer, ClientOrder> oldOrders = new HashMap<>();
        for (ClientOrder order: orders) {
            oldOrders.put(order.getId(), order.deepCopy());
        }

        // Update orders
        List<ClientOrder> updatedOrders = orders.stream()
                .map(order -> {
                    UpdateClientOrderDTO orderDTO = orderDTOs.stream()
                            .filter(dto -> dto.getId().equals(order.getId()))
                            .findFirst()
                            .orElseThrow(() -> new ResourceNotFoundException("Client Order with ID: " + order.getId() + " not found."));
                    UpdateClientOrderDTO sanitizedOrderDTO = entitySanitizerService.sanitizeUpdateClientOrderDTO(orderDTO);
                    ClientDTOMapper.setUpdateClientOrderDTOToUpdateOrder(order, sanitizedOrderDTO);
                    Component component = componentRepository.findById(sanitizedOrderDTO.getComponentId())
                            .orElseThrow(() -> new ResourceNotFoundException("Component with ID: " + sanitizedOrderDTO.getComponentId() + " not found."));
                    order.setComponentId(component.getId());
                    return order;
                }).toList();

        // Save
        List<ClientOrder> savedOrders = clientOrderRepository.saveAll(updatedOrders);

        // Publish order events to Kafka broker
        List<ClientOrderEvent> orderEvents = new ArrayList<>();
        savedOrders.stream()
                .map(order -> {
                    ClientOrder oldOrder = oldOrders.get(order.getId());
                    return new ClientOrderEvent(order, oldOrder, KafkaEvent.EventType.UPDATE, order.getClientId(), Feature.SUPPLIER, "Test");
                })
                .forEach(orderEvents::add);

        kafkaClientOrderService.sendClientOrderEventsInBulk(orderEvents);

        return savedOrders;
    }

    @Transactional
    public List<Integer> deleteClientOrdersInBulk(List<Integer> orderIds) {
        List<ClientOrder> orders = clientOrderRepository.findAllById(orderIds);
        // Ensure same organizationId
        if (orders.stream().map(ClientOrder::getOrganizationId).distinct().count() > 1) {
            throw new ValidationException("All orders must belong to the same organization.");
        }

        clientOrderRepository.deleteAll(orders);

        // Publish order events to Kafka broker
        List<ClientOrderEvent> orderEvents = new ArrayList<>();
        orders.stream()
                .map(order -> new ClientOrderEvent(null, order, KafkaEvent.EventType.CREATE, order.getClientId(), Feature.SUPPLIER, "Test"))
                .forEach(orderEvents::add);

        kafkaClientOrderService.sendClientOrderEventsInBulk(orderEvents);

        return orderIds;
    }

}
