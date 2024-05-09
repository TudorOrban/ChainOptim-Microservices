package org.chainoptimnotifications.core.notification.service;

import jakarta.transaction.Transactional;
import org.apache.kafka.common.errors.ResourceNotFoundException;
import org.chainoptimnotifications.core.notification.dto.AddNotificationDTO;
import org.chainoptimnotifications.core.notification.dto.NotificationDTOMapper;
import org.chainoptimnotifications.core.notification.dto.UpdateNotificationDTO;
import org.chainoptimnotifications.core.notification.model.Notification;
import org.chainoptimnotifications.core.notification.model.NotificationUser;
import org.chainoptimnotifications.core.notification.repository.NotificationRepository;
import org.chainoptimnotifications.core.notification.repository.NotificationUserRepository;
import org.chainoptimnotifications.shared.search.model.PaginatedResults;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class NotificationPersistenceServiceImpl implements NotificationPersistenceService {

    private final NotificationRepository notificationRepository;
    private final NotificationUserRepository notificationUserRepository;
    private final JdbcTemplate jdbcTemplate;

    @Autowired
    public NotificationPersistenceServiceImpl(NotificationRepository notificationRepository,
                                              NotificationUserRepository notificationUserRepository,
                                              JdbcTemplate jdbcTemplate) {
        this.notificationRepository = notificationRepository;
        this.notificationUserRepository = notificationUserRepository;
        this.jdbcTemplate = jdbcTemplate;
    }

    public List<NotificationUser> getNotificationsByUserId(String userId) {
        return notificationUserRepository.findByUserId(userId);
    }

    public PaginatedResults<NotificationUser> getNotificationsByUserIdAdvanced(String userId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage) {
        return notificationUserRepository.findByUserIdAdvanced(userId, searchQuery, sortBy, ascending, page, itemsPerPage);
    }

    /*
     * Use JdbcTemplate for this particular operation
     * as usual .save causes StackOverflow for some reason
     */
    @Transactional
    public Notification addNotification(AddNotificationDTO notificationDTO) {
        Notification notification = notificationRepository.save(NotificationDTOMapper.mapAddNotificationDTOToNotification(notificationDTO));

        List<Object[]> batchArgs = new ArrayList<>();
        for (String userId : notificationDTO.getUserIds()) {
            Object[] values = {
                    notification.getId(),
                    false,
                    userId
            };
            batchArgs.add(values);
        }

        jdbcTemplate.batchUpdate(
                "INSERT INTO notification_users (notification_id, read_status, user_id) VALUES (?, ?, ?)",
                batchArgs
        );

        return notification;
    }


    public Notification updateNotification(UpdateNotificationDTO notificationDTO) {
        Notification notification = notificationRepository.findById(notificationDTO.getId())
                .orElseThrow(() -> new ResourceNotFoundException("Notification with ID: " + notificationDTO.getId() + " not found"));

        Notification updatedNotification = NotificationDTOMapper.setUpdateNotificationDTOToNotification(notificationDTO, notification);

        return notificationRepository.save(updatedNotification);
    }

    public void deleteNotification(Integer notificationId) {
        notificationRepository.deleteById(notificationId);
    }
}
