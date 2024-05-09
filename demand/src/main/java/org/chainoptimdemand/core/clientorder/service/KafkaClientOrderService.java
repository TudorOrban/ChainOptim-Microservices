package org.chainoptimdemand.core.clientorder.service;


import org.chainoptimdemand.core.clientorder.model.ClientOrderEvent;

import java.util.List;

public interface KafkaClientOrderService {

    void sendClientOrderEvent(ClientOrderEvent orderEvent);
    void sendClientOrderEventsInBulk(List<ClientOrderEvent> kafkaEvents);
}
