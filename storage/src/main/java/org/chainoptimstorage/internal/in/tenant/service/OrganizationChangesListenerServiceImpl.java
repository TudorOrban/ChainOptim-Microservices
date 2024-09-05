package org.chainoptimstorage.internal.in.tenant.service;

import org.chainoptimstorage.shared.kafka.KafkaEvent;
import org.chainoptimstorage.shared.kafka.OrganizationEvent;
import org.chainoptimstorage.core.warehouse.model.Warehouse;
import org.chainoptimstorage.core.warehouse.repository.WarehouseRepository;
import org.chainoptimstorage.core.supplierorder.model.SupplierOrder;
import org.chainoptimstorage.core.supplierorder.repository.SupplierOrderRepository;
import org.chainoptimstorage.core.suppliershipment.model.SupplierShipment;
import org.chainoptimstorage.core.suppliershipment.repository.SupplierShipmentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class OrganizationChangesListenerServiceImpl implements OrganizationChangesListenerService {

    private final WarehouseRepository warehouseRepository;
    private final SupplierOrderRepository supplierOrderRepository;
    private final SupplierShipmentRepository supplierShipmentRepository;

    @Autowired
    public OrganizationChangesListenerServiceImpl(WarehouseRepository warehouseRepository,
                                                  SupplierOrderRepository supplierOrderRepository,
                                                  SupplierShipmentRepository supplierShipmentRepository) {
        this.warehouseRepository = warehouseRepository;
        this.supplierOrderRepository = supplierOrderRepository;
        this.supplierShipmentRepository = supplierShipmentRepository;
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

        List<SupplierOrder> supplierOrders = supplierOrderRepository.findByOrganizationId(organizationId);
        supplierOrderRepository.deleteAll(supplierOrders);

        List<Integer> supplierOrderIds = supplierOrders.stream()
                .map(SupplierOrder::getId).toList();
        List<SupplierShipment> supplierShipments = supplierShipmentRepository.findBySupplierOrderIds(supplierOrderIds);
        supplierShipmentRepository.deleteAll(supplierShipments);
    }
}
