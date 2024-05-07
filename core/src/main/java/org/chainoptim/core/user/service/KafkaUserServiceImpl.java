package org.chainoptim.core.user.service;


import org.chainoptim.shared.kafka.UserEvent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class KafkaUserServiceImpl implements KafkaUserService {

    private final KafkaTemplate<String, UserEvent> kafkaTemplate;

    @Autowired
    public KafkaUserServiceImpl(KafkaTemplate<String, UserEvent> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    @Value("${user.topic.name:user-events}")
    private String userTopicName;

    public void sendUserEvent(UserEvent userEvent) {
        kafkaTemplate.send(userTopicName, userEvent);
    }

    public void sendUserEventsInBulk(List<UserEvent> kafkaEvents) {
        kafkaEvents
                .forEach(userEvent -> kafkaTemplate.send(userTopicName, userEvent));
    }
}
