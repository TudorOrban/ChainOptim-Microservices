package org.chainoptimstorage.core.warehouseinventoryitem.model;

import org.chainoptimstorage.shared.enums.Feature;
import org.chainoptimstorage.shared.kafka.KafkaEvent;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class WarehouseInventoryEvent extends KafkaEvent<WarehouseInventoryItem> {

    public WarehouseInventoryEvent(WarehouseInventoryItem newEntity, WarehouseInventoryItem oldEntity, EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.SUPPLIER_ORDER, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}
