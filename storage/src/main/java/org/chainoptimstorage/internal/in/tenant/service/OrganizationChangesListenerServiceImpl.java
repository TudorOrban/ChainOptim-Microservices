package org.chainoptimstorage.internal.in.tenant.service;

import org.chainoptimstorage.shared.kafka.KafkaEvent;
import org.chainoptimstorage.shared.kafka.OrganizationEvent;
import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.core.warehouse.repository.WarehouseRepository;
import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryItem;
import org.chainoptimstorage.core.warehouseinventoryitem.repository.WarehouseInventoryItemRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class OrganizationChangesListenerServiceImpl implements OrganizationChangesListenerService {

    private final WarehouseRepository warehouseRepository;
    private final WarehouseInventoryItemRepository warehouseInventoryItemRepository;

    @Autowired
    public OrganizationChangesListenerServiceImpl(WarehouseRepository warehouseRepository,
                                                  WarehouseInventoryItemRepository warehouseInventoryItemRepository) {
        this.warehouseRepository = warehouseRepository;
        this.warehouseInventoryItemRepository = warehouseInventoryItemRepository;
    }

    @KafkaListener(topics = "${organization.topic.name:organization-events}", groupId = "supply-organization-group", containerFactory = "organizationKafkaListenerContainerFactory")
    public void listenToOrganizationEvent(OrganizationEvent event) {
        System.out.println("Organization Event in Supply: " + event);
        if (event.getEventType() != KafkaEvent.EventType.DELETE) return;

        cleanUpOrganizationEntities(event.getOldEntity().getId());
    }

    private void cleanUpOrganizationEntities(Integer organizationId) {
        List<Warehouse> warehouses = warehouseRepository.findByOrganizationId(organizationId);
        warehouseRepository.deleteAll(warehouses);

        List<WarehouseInventoryItem> warehouseInventoryItems = warehouseInventoryItemRepository.findByOrganizationId(organizationId);
        warehouseInventoryItemRepository.deleteAll(warehouseInventoryItems);
    }
}
