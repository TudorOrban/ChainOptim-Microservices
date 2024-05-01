package org.chainoptimnotifications.notification.repository;

import org.chainoptimnotifications.notification.model.Notification;
import org.springframework.data.jpa.repository.JpaRepository;

public interface NotificationRepository extends JpaRepository<Notification, Integer> {

}
