package org.chainoptimnotifications.core.notification.repository;

import org.chainoptimnotifications.core.notification.model.NotificationUser;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface NotificationUserRepository extends JpaRepository<NotificationUser, Integer>, NotificationSearchRepository {

    List<NotificationUser> findByUserId(String userId);

    @Query("SELECT nu FROM NotificationUser nu WHERE nu.userId = :userId")
    List<NotificationUser> findAllByUserId(@Param("userId") String userId);
}
