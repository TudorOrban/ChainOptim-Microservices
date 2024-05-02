package org.chainoptimnotifications.notification.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.chainoptimnotifications.email.service.EmailService;
import org.chainoptimnotifications.notification.dto.AddNotificationDTO;
import org.chainoptimnotifications.notification.dto.NotificationDTOMapper;
import org.chainoptimnotifications.notification.model.Notification;
import org.chainoptimnotifications.notification.model.NotificationUserDistribution;
import org.chainoptimnotifications.notification.websocket.WebSocketMessagingService;
import org.chainoptimnotifications.outsidefeatures.model.ClientOrderEvent;
import org.chainoptimnotifications.outsidefeatures.model.SupplierOrderEvent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class NotificationServiceImpl implements NotificationService {

    private final WebSocketMessagingService messagingService;
    private final NotificationFormatterService notificationFormatterService;
    private final NotificationPersistenceService notificationPersistenceService;
    private final NotificationDistributionService notificationDistributionService;
    private final EmailService emailService;
    private final ObjectMapper objectMapper;

    @Value("${app.environment}")
    private String appEnvironment;

    @Autowired
    public NotificationServiceImpl(WebSocketMessagingService messagingService,
                                   NotificationFormatterService notificationFormatterService,
                                   NotificationPersistenceService notificationPersistenceService,
                                   NotificationDistributionService notificationDistributionService,
                                   EmailService emailService,
                                   ObjectMapper objectMapper) {
        this.messagingService = messagingService;
        this.notificationFormatterService = notificationFormatterService;
        this.notificationPersistenceService = notificationPersistenceService;
        this.notificationDistributionService = notificationDistributionService;
        this.emailService = emailService;
        this.objectMapper = objectMapper;
    }

    public void createNotification(SupplierOrderEvent event) {
        Notification notification = notificationFormatterService.formatEvent(event);

        NotificationUserDistribution users = notificationDistributionService.distributeEventToUsers(event);

        sendNotification(notification, users);
    }

    public void createNotification(ClientOrderEvent event) {
        Notification notification = notificationFormatterService.formatEvent(event);

        NotificationUserDistribution users = notificationDistributionService.distributeEventToUsers(event);

        sendNotification(notification, users);
    }

    private void sendNotification(Notification notification, NotificationUserDistribution users) {
        List<String> notificationUserIds = users.getNotificationUserIds();
        List<String> emailUserEmails = users.getEmailUserEmails();

        sendRealTimeNotification(notification, notificationUserIds);

        if ("prod".equals(appEnvironment)) { // Only send emails in production
            sendEmailNotification(notification, emailUserEmails);
        }

        // Persist the notification in the database
        AddNotificationDTO notificationDTO = NotificationDTOMapper.mapNotificationToAddNotificationDTO(notification, notificationUserIds);
        notificationDTO.setUserIds(notificationUserIds);
        notificationPersistenceService.addNotification(notificationDTO);
    }

    private void sendRealTimeNotification(Notification notification, List<String> userIds) {
        System.out.println("Sending notification: " + notification);
        try {
            String serializedNotification = objectMapper.writeValueAsString(notification);

            // Send the event to all users connected to the WebSocket
            System.out.println("Sessions: " + messagingService.getSessions());
            for (String userId : userIds) {
                if (messagingService.getSessions().containsKey(userId)) {
                    messagingService.sendMessageToUser(userId, serializedNotification);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void sendEmailNotification(Notification notification, List<String> userEmails) {
        System.out.println("Sending email notification: " + notification);
        for (String userEmail : userEmails) {
            emailService.sendEmail(userEmail, notification.getTitle(), notification.getMessage());
        }
    }
}
