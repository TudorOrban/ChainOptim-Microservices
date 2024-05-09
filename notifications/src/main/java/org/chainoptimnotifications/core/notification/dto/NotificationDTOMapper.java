package org.chainoptimnotifications.core.notification.dto;


import org.chainoptimnotifications.core.notification.model.Notification;

import java.util.List;

public class NotificationDTOMapper {


    public static Notification mapAddNotificationDTOToNotification(AddNotificationDTO addNotificationDTO) {
        Notification notification = new Notification();
        notification.setTitle(addNotificationDTO.getTitle());
        notification.setEntityId(addNotificationDTO.getEntityId());
        notification.setEntityType(addNotificationDTO.getEntityType());
        notification.setMessage(addNotificationDTO.getMessage());
        notification.setType(addNotificationDTO.getType());

        return notification;
    }

    public static AddNotificationDTO mapNotificationToAddNotificationDTO(Notification notification, List<String> userIds) {
        AddNotificationDTO addNotificationDTO = new AddNotificationDTO();
        addNotificationDTO.setTitle(notification.getTitle());
        addNotificationDTO.setEntityId(notification.getEntityId());
        addNotificationDTO.setEntityType(notification.getEntityType());
        addNotificationDTO.setMessage(notification.getMessage());
        addNotificationDTO.setType(notification.getType());
        addNotificationDTO.setUserIds(userIds);

        return addNotificationDTO;
    }

    public static Notification setUpdateNotificationDTOToNotification(UpdateNotificationDTO updateNotificationDTO, Notification notification) {
        notification.setTitle(updateNotificationDTO.getTitle());
        notification.setEntityId(updateNotificationDTO.getEntityId());
        notification.setEntityType(updateNotificationDTO.getEntityType());
        notification.setMessage(updateNotificationDTO.getMessage());
        notification.setType(updateNotificationDTO.getType());

        return notification;
    }
}
