package org.chainoptimstorage.core.warehouseinventoryitem.service;


import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryEvent;

import java.util.List;

public interface KafkaWarehouseInventoryItemService {

    void sendWarehouseInventoryItemEvent(WarehouseInventoryEvent itemEvent);
    void sendWarehouseInventoryItemEventsInBulk(List<WarehouseInventoryEvent> kafkaEvents);
}
