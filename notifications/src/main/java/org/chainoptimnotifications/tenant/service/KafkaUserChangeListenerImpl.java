package org.chainoptimnotifications.tenant.service;

import org.chainoptimnotifications.notification.model.KafkaEvent;
import org.chainoptimnotifications.notification.model.NotificationUser;
import org.chainoptimnotifications.notification.repository.NotificationUserRepository;
import org.chainoptimnotifications.outsidefeatures.model.UserEvent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class KafkaUserChangeListenerImpl implements KafkaUserChangeListener {

    private final NotificationUserRepository notificationUserRepository;

    @Autowired
    public KafkaUserChangeListenerImpl(NotificationUserRepository notificationUserRepository) {
        this.notificationUserRepository = notificationUserRepository;
    }

    @KafkaListener(topics = "${user.topic.name:user-events}", groupId = "user-group", containerFactory = "userKafkaListenerContainerFactory")
    public void listenUserEvent(UserEvent event) {
        System.out.println("User Event Test: " + event);
        if (event.getEventType() != KafkaEvent.EventType.DELETE) return;

        List<NotificationUser> notificationUsers = notificationUserRepository.findByUserId(event.getOldEntity().getId());

        notificationUserRepository.deleteAll(notificationUsers);
    }
}
