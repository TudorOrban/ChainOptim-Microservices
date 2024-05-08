package org.chainoptimsupply.supplierorder.service;


import org.chainoptimsupply.supplierorder.model.SupplierOrderEvent;

import java.util.List;

public interface KafkaSupplierOrderService {

    void sendSupplierOrderEvent(SupplierOrderEvent orderEvent);
    void sendSupplierOrderEventsInBulk(List<SupplierOrderEvent> kafkaEvents);
}
