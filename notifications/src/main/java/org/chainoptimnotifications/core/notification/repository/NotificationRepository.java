package org.chainoptimnotifications.core.notification.repository;

import org.chainoptimnotifications.core.notification.model.Notification;
import org.springframework.data.jpa.repository.JpaRepository;

public interface NotificationRepository extends JpaRepository<Notification, Integer> {

}
