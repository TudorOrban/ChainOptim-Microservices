package org.chainoptimnotifications.notification.service;

import org.chainoptimnotifications.notification.dto.AddNotificationDTO;
import org.chainoptimnotifications.notification.dto.UpdateNotificationDTO;
import org.chainoptimnotifications.notification.model.Notification;
import org.chainoptimnotifications.notification.model.NotificationUser;
import org.chainoptimnotifications.outsidefeatures.model.PaginatedResults;

import java.util.List;

public interface NotificationPersistenceService {

    List<NotificationUser> getNotificationsByUserId(String userId);
    PaginatedResults<NotificationUser> getNotificationsByUserIdAdvanced(String userId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage);

    Notification addNotification(AddNotificationDTO notification);
    Notification updateNotification(UpdateNotificationDTO notification);
    void deleteNotification(Integer notificationId);
}
