package org.chainoptimstorage.core.warehouseinventoryitem.service;

import org.chainoptimstorage.core.warehouse.dto.WarehouseDTOMapper;
import org.chainoptimstorage.core.warehouseinventoryitem.dto.CreateWarehouseInventoryItemDTO;
import org.chainoptimstorage.core.warehouseinventoryitem.dto.UpdateWarehouseInventoryItemDTO;
import org.chainoptimstorage.core.warehouseinventoryitem.dto.WarehouseInventoryItemDTOMapper;
import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryItem;
import org.chainoptimstorage.exception.PlanLimitReachedException;
import org.chainoptimstorage.exception.ResourceNotFoundException;
import org.chainoptimstorage.exception.ValidationException;
import org.chainoptimstorage.internal.in.goods.repository.ComponentRepository;
import org.chainoptimstorage.internal.in.goods.model.Component;
import org.chainoptimstorage.internal.in.tenant.service.SubscriptionPlanLimiterService;
import org.chainoptimstorage.shared.enums.Feature;
import org.chainoptimstorage.shared.kafka.KafkaEvent;
import org.chainoptimstorage.shared.PaginatedResults;
import org.chainoptimstorage.shared.enums.SearchMode;
import org.chainoptimstorage.shared.sanitization.EntitySanitizerService;
import org.chainoptimstorage.shared.search.SearchParams;
import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryEvent;
import org.chainoptimstorage.core.warehouseinventoryitem.repository.WarehouseInventoryItemRepository;

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
public class WarehouseInventoryItemServiceImpl implements WarehouseInventoryItemService {

    private final WarehouseInventoryItemRepository warehouseInventoryItemRepository;
    private final KafkaWarehouseInventoryItemService kafkaWarehouseInventoryItemService;
    private final ComponentRepository componentRepository;
    private final SubscriptionPlanLimiterService planLimiterService;
    private final EntitySanitizerService entitySanitizerService;

    @Autowired
    public WarehouseInventoryItemServiceImpl(
            WarehouseInventoryItemRepository warehouseInventoryItemRepository,
            KafkaWarehouseInventoryItemService kafkaWarehouseInventoryItemService,
            ComponentRepository componentRepository,
            SubscriptionPlanLimiterService planLimiterService,
            EntitySanitizerService entitySanitizerService
    ) {
        this.warehouseInventoryItemRepository = warehouseInventoryItemRepository;
        this.kafkaWarehouseInventoryItemService = kafkaWarehouseInventoryItemService;
        this.componentRepository = componentRepository;
        this.planLimiterService = planLimiterService;
        this.entitySanitizerService = entitySanitizerService;
    }

    public List<WarehouseInventoryItem> getWarehouseInventoryItemsByOrganizationId(Integer organizationId) {
        return warehouseInventoryItemRepository.findByOrganizationId(organizationId);
    }

    public List<WarehouseInventoryItem> getWarehouseInventoryItemsByWarehouseId(Integer warehouseId) {
        return warehouseInventoryItemRepository.findByWarehouseId(warehouseId);
    }

    public PaginatedResults<WarehouseInventoryItem> getWarehouseInventoryItemsAdvanced(SearchMode searchMode, Integer entityId, SearchParams searchParams) {
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

        return warehouseInventoryItemRepository.findByWarehouseIdAdvanced(searchMode, entityId, searchParams);
    }

    public Integer getOrganizationIdById(Long warehouseInventoryItemId) {
        return warehouseInventoryItemRepository.findOrganizationIdById(warehouseInventoryItemId)
                .orElseThrow(() -> new ResourceNotFoundException("Warehouse Item with ID: " + warehouseInventoryItemId + " not found."));
    }

    public long countByOrganizationId(Integer organizationId) {
        return warehouseInventoryItemRepository.countByOrganizationId(organizationId);
    }

    // Create
    public WarehouseInventoryItem createWarehouseInventoryItem(CreateWarehouseInventoryItemDTO itemDTO) {
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(itemDTO.getOrganizationId(), Feature.SUPPLIER_ORDER, 1)) {
            throw new PlanLimitReachedException("You have reached the limit of allowed Warehouse Items for the current Subscription Plan.");
        }

        // Sanitize input and map to entity
        CreateWarehouseInventoryItemDTO sanitizedItemDTO = entitySanitizerService.sanitizeCreateWarehouseInventoryItemDTO(itemDTO);
        WarehouseInventoryItem warehouseInventoryItem = WarehouseInventoryItemDTOMapper.mapCreateDtoToWarehouseInventoryItem(sanitizedItemDTO);
        Component component = componentRepository.findById(sanitizedItemDTO.getComponentId())
                .orElseThrow(() -> new ResourceNotFoundException("Component with ID: " + sanitizedItemDTO.getComponentId() + " not found."));
        warehouseInventoryItem.setComponent(component);

        WarehouseInventoryItem savedItem = warehouseInventoryItemRepository.save(warehouseInventoryItem);

        // Publish item to Kafka broker
        kafkaWarehouseInventoryItemService.sendWarehouseInventoryItemEvent(
                new WarehouseInventoryEvent(savedItem, null, KafkaEvent.EventType.CREATE, savedItem.getWarehouseId(), Feature.SUPPLIER, "Test"));

        return savedItem;
    }

    @Transactional
    public List<WarehouseInventoryItem> createWarehouseInventoryItemsInBulk(List<CreateWarehouseInventoryItemDTO> itemDTOs) {
        // Ensure same organizationId
        if (itemDTOs.stream().map(CreateWarehouseInventoryItemDTO::getOrganizationId).distinct().count() > 1) {
            throw new ValidationException("All items must belong to the same organization.");
        }
        // Check if plan limit is reached
        if (planLimiterService.isLimitReached(itemDTOs.getFirst().getOrganizationId(), Feature.SUPPLIER_ORDER, itemDTOs.size())) {
            throw new PlanLimitReachedException("You have reached the limit of allowed Warehouse Items for the current Subscription Plan.");
        }

        // Sanitize and map to entity
        List<WarehouseInventoryItem> items = itemDTOs.stream()
                .map(itemDTO -> {
                    CreateWarehouseInventoryItemDTO sanitizedItemDTO = entitySanitizerService.sanitizeCreateWarehouseInventoryItemDTO(itemDTO);
                    WarehouseInventoryItem warehouseInventoryItem = WarehouseInventoryItemDTOMapper.mapCreateDtoToWarehouseInventoryItem(sanitizedItemDTO);
                    Component component = componentRepository.findById(sanitizedItemDTO.getComponentId())
                            .orElseThrow(() -> new ResourceNotFoundException("Component with ID: " + sanitizedItemDTO.getComponentId() + " not found."));
                    warehouseInventoryItem.setComponent(component);
                    return warehouseInventoryItem;
                })
                .toList();

        List<WarehouseInventoryItem> savedItems = warehouseInventoryItemRepository.saveAll(items);

        // Publish item events to Kafka broker
        List<WarehouseInventoryEvent> itemEvents = new ArrayList<>();
        items.stream()
                .map(item -> new WarehouseInventoryEvent(item, null, KafkaEvent.EventType.CREATE, item.getWarehouseId(), Feature.SUPPLIER, "Test"))
                .forEach(itemEvents::add);

        kafkaWarehouseInventoryItemService.sendWarehouseInventoryItemEventsInBulk(itemEvents);

        return savedItems;
    }

    @Transactional
    public List<WarehouseInventoryItem> updateWarehouseInventoryItemsInBulk(List<UpdateWarehouseInventoryItemDTO> itemDTOs) {
        List<WarehouseInventoryItem> items = warehouseInventoryItemRepository.findByIds(itemDTOs.stream().map(UpdateWarehouseInventoryItemDTO::getId).toList())
                .orElseThrow(() -> new ResourceNotFoundException("Warehouse Items not found."));

        // Save old items for event publishing
        Map<Integer, WarehouseInventoryItem> oldItems = new HashMap<>();
        for (WarehouseInventoryItem item: items) {
            oldItems.put(item.getId(), item.deepCopy());
        }

        // Update items
        List<WarehouseInventoryItem> updatedItems = items.stream()
                .map(item -> {
                    UpdateWarehouseInventoryItemDTO itemDTO = itemDTOs.stream()
                            .filter(dto -> dto.getId().equals(item.getId()))
                            .findFirst()
                            .orElseThrow(() -> new ResourceNotFoundException("Warehouse Item with ID: " + item.getId() + " not found."));
                    UpdateWarehouseInventoryItemDTO sanitizedItemDTO = entitySanitizerService.sanitizeUpdateWarehouseInventoryItemDTO(itemDTO);
                    WarehouseInventoryItemDTOMapper.setUpdateWarehouseInventoryItemDTOToUpdateInventoryItem(item, sanitizedItemDTO);
                    Component component = componentRepository.findById(sanitizedItemDTO.getComponentId())
                            .orElseThrow(() -> new ResourceNotFoundException("Component with ID: " + sanitizedItemDTO.getComponentId() + " not found."));
                    item.setComponent(component);
                    return item;
                }).toList();

        // Save
        List<WarehouseInventoryItem> savedItems = warehouseInventoryItemRepository.saveAll(updatedItems);

        // Publish item events to Kafka broker
        List<WarehouseInventoryEvent> itemEvents = new ArrayList<>();
        savedItems.stream()
                .map(item -> {
                    WarehouseInventoryItem oldItem = oldItems.get(item.getId());
                    return new WarehouseInventoryEvent(item, oldItem, KafkaEvent.EventType.UPDATE, item.getWarehouseId(), Feature.SUPPLIER, "Test");
                })
                .forEach(itemEvents::add);

        kafkaWarehouseInventoryItemService.sendWarehouseInventoryItemEventsInBulk(itemEvents);

        return savedItems;
    }

    @Transactional
    public List<Integer> deleteWarehouseInventoryItemsInBulk(List<Integer> itemIds) {
        List<WarehouseInventoryItem> items = warehouseInventoryItemRepository.findAllById(itemIds);
        // Ensure same organizationId
        if (items.stream().map(WarehouseInventoryItem::getOrganizationId).distinct().count() > 1) {
            throw new ValidationException("All items must belong to the same organization.");
        }

        warehouseInventoryItemRepository.deleteAll(items);

        // Publish item events to Kafka broker
        List<WarehouseInventoryEvent> itemEvents = new ArrayList<>();
        items.stream()
                .map(item -> new WarehouseInventoryEvent(null, item, KafkaEvent.EventType.CREATE, item.getWarehouseId(), Feature.SUPPLIER, "Test"))
                .forEach(itemEvents::add);

        kafkaWarehouseInventoryItemService.sendWarehouseInventoryItemEventsInBulk(itemEvents);

        return itemIds;
    }

}
