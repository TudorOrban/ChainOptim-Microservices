package org.chainoptim.internalcommunication.in.storage.model;

import org.chainoptim.shared.enums.Feature;
import org.chainoptim.shared.kafka.KafkaEvent;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class WarehouseInventoryEvent extends KafkaEvent<WarehouseInventoryItem> {

    public WarehouseInventoryEvent(WarehouseInventoryItem newEntity, WarehouseInventoryItem oldEntity, EventType eventType, Integer mainEntityId, Feature mainEntityType, String mainEntityName) {
        super(newEntity, oldEntity, Feature.WAREHOUSE_INVENTORY, eventType, mainEntityId, mainEntityType, mainEntityName);
    }
}
