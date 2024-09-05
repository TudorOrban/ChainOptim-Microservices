package org.chainoptimstorage.core.warehouseinventoryitem.service;

import org.chainoptimstorage.core.warehouseinventoryitem.model.WarehouseInventoryEvent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class KafkaWarehouseInventoryItemServiceImpl implements KafkaWarehouseInventoryItemService {

    private final KafkaTemplate<String, WarehouseInventoryEvent> kafkaTemplate;

    @Autowired
    public KafkaWarehouseInventoryItemServiceImpl(KafkaTemplate<String, WarehouseInventoryEvent> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    @Value("${warehouse.inventory.item.topic.name:warehouse-inventory-item-events}")
    private String warehouseInventoryItemTopicName;

    public void sendWarehouseInventoryItemEvent(WarehouseInventoryEvent itemEvent) {
        kafkaTemplate.send(warehouseInventoryItemTopicName, itemEvent);
    }

    public void sendWarehouseInventoryItemEventsInBulk(List<WarehouseInventoryEvent> kafkaEvents) {
        kafkaEvents
                .forEach(itemEvent -> kafkaTemplate.send(warehouseInventoryItemTopicName, itemEvent));
    }
}
