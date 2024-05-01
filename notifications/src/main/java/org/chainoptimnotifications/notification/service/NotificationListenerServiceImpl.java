package org.chainoptimnotifications.notification.service;

import org.chainoptimnotifications.outsidefeatures.model.ClientOrderEvent;
import org.chainoptimnotifications.outsidefeatures.model.SupplierOrderEvent;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

@Service
public class NotificationListenerServiceImpl implements NotificationListenerService {

    private final NotificationService notificationService;

    @Autowired
    public NotificationListenerServiceImpl(NotificationService notificationService) {
        this.notificationService = notificationService;
    }

    @KafkaListener(topics = "supplier-order-events", groupId = "notification-group", containerFactory = "supplierOrderKafkaListenerContainerFactory")
    public void listenSupplierOrderEvent(SupplierOrderEvent event) {
        System.out.println("Supplier Order Event: " + event);
        notificationService.createNotification(event);
    }

    @KafkaListener(topics = "client-order-events", groupId = "notification-group", containerFactory = "clientOrderKafkaListenerContainerFactory")
    public void listenClientOrderEvent(ClientOrderEvent event) {
        System.out.println("Client Order Event: " + event);
        notificationService.createNotification(event);

    }
}
