package org.chainoptimnotifications.user.service;

import org.chainoptimnotifications.outsidefeatures.model.UserEvent;
import org.springframework.kafka.annotation.KafkaListener;

public class KafkaUserChangeListenerImpl implements KafkaUserChangeListener {

    @KafkaListener(topics = "user-events", groupId = "user-group", containerFactory = "userKafkaListenerContainerFactory")
    public void listenUserEvent(UserEvent event) {
        System.out.println("User Event Test: " + event);
    }
}
