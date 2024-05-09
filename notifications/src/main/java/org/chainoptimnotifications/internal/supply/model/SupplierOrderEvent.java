package org.chainoptimnotifications.internal.supply.model;

import lombok.AllArgsConstructor;
import org.chainoptimnotifications.shared.enums.Feature;
import org.chainoptimnotifications.core.notification.model.KafkaEvent;

@AllArgsConstructor
public class SupplierOrderEvent extends KafkaEvent<SupplierOrder> {

    public SupplierOrderEvent(SupplierOrder newEntity, SupplierOrder oldEntity, EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.SUPPLIER_ORDER, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}
