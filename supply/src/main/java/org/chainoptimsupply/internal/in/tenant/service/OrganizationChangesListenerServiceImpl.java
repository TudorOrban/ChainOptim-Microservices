package org.chainoptimsupply.internal.in.tenant.service;

import org.chainoptimsupply.shared.kafka.KafkaEvent;
import org.chainoptimsupply.shared.kafka.OrganizationEvent;
import org.chainoptimsupply.core.supplier.model.Supplier;
import org.chainoptimsupply.core.supplier.repository.SupplierRepository;
import org.chainoptimsupply.core.supplierorder.model.SupplierOrder;
import org.chainoptimsupply.core.supplierorder.repository.SupplierOrderRepository;
import org.chainoptimsupply.core.suppliershipment.model.SupplierShipment;
import org.chainoptimsupply.core.suppliershipment.repository.SupplierShipmentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class OrganizationChangesListenerServiceImpl implements OrganizationChangesListenerService {

    private final SupplierRepository supplierRepository;
    private final SupplierOrderRepository supplierOrderRepository;
    private final SupplierShipmentRepository supplierShipmentRepository;

    @Autowired
    public OrganizationChangesListenerServiceImpl(SupplierRepository supplierRepository,
                                                  SupplierOrderRepository supplierOrderRepository,
                                                  SupplierShipmentRepository supplierShipmentRepository) {
        this.supplierRepository = supplierRepository;
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
        List<Supplier> suppliers = supplierRepository.findByOrganizationId(organizationId);
        supplierRepository.deleteAll(suppliers);

        List<SupplierOrder> supplierOrders = supplierOrderRepository.findByOrganizationId(organizationId);
        supplierOrderRepository.deleteAll(supplierOrders);

        List<Integer> supplierOrderIds = supplierOrders.stream()
                .map(SupplierOrder::getId).toList();
        List<SupplierShipment> supplierShipments = supplierShipmentRepository.findBySupplierOrderIds(supplierOrderIds);
        supplierShipmentRepository.deleteAll(supplierShipments);
    }
}
