package org.chainoptimsupply.core.supplierorder.service;


import org.chainoptimsupply.core.supplierorder.model.SupplierOrderEvent;

import java.util.List;

public interface KafkaSupplierOrderService {

    void sendSupplierOrderEvent(SupplierOrderEvent orderEvent);
    void sendSupplierOrderEventsInBulk(List<SupplierOrderEvent> kafkaEvents);
}
