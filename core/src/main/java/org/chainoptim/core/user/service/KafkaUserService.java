package org.chainoptim.core.user.service;

import org.chainoptim.shared.kafka.UserEvent;

import java.util.List;

public interface KafkaUserService {

    void sendUserEvent(UserEvent userEvent);
    void sendUserEventsInBulk(List<UserEvent> kafkaEvents);
}
