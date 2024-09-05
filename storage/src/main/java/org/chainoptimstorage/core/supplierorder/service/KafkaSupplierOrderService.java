package org.chainoptimstorage.core.supplierorder.service;


import org.chainoptimstorage.core.supplierorder.model.SupplierOrderEvent;

import java.util.List;

public interface KafkaSupplierOrderService {

    void sendSupplierOrderEvent(SupplierOrderEvent orderEvent);
    void sendSupplierOrderEventsInBulk(List<SupplierOrderEvent> kafkaEvents);
}
