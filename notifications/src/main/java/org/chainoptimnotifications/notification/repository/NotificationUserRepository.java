package org.chainoptimnotifications.notification.repository;

import org.chainoptimnotifications.notification.model.NotificationUser;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface NotificationUserRepository extends JpaRepository<NotificationUser, Integer>, NotificationSearchRepository {

    List<NotificationUser> findByUserId(String userId);
}
