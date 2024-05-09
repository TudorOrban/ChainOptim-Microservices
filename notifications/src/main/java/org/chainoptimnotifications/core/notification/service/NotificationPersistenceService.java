package org.chainoptimnotifications.core.notification.service;

import org.chainoptimnotifications.core.notification.dto.AddNotificationDTO;
import org.chainoptimnotifications.core.notification.dto.UpdateNotificationDTO;
import org.chainoptimnotifications.core.notification.model.Notification;
import org.chainoptimnotifications.core.notification.model.NotificationUser;
import org.chainoptimnotifications.shared.search.model.PaginatedResults;

import java.util.List;

public interface NotificationPersistenceService {

    List<NotificationUser> getNotificationsByUserId(String userId);
    PaginatedResults<NotificationUser> getNotificationsByUserIdAdvanced(String userId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage);

    Notification addNotification(AddNotificationDTO notification);
    Notification updateNotification(UpdateNotificationDTO notification);
    void deleteNotification(Integer notificationId);
}
