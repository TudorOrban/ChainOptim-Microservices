package org.chainoptim.internalcommunication.in.supplier.model;

import org.chainoptim.shared.enums.Feature;
import org.chainoptim.shared.kafka.KafkaEvent;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class SupplierOrderEvent extends KafkaEvent<SupplierOrder> {

    public SupplierOrderEvent(SupplierOrder newEntity, SupplierOrder oldEntity, EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.SUPPLIER_ORDER, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}
