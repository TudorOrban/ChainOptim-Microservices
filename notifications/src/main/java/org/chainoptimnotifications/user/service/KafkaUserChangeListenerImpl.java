package org.chainoptimnotifications.user.service;

import org.chainoptimnotifications.outsidefeatures.model.UserEvent;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

@Service
public class KafkaUserChangeListenerImpl implements KafkaUserChangeListener {

    @KafkaListener(topics = "${user.topic.name:user-events}", groupId = "user-group", containerFactory = "userKafkaListenerContainerFactory")
    public void listenUserEvent(UserEvent event) {
        System.out.println("User Event Test: " + event);
    }
}
