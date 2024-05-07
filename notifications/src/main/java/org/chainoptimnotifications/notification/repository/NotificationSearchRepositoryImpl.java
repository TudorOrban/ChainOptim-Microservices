package org.chainoptimnotifications.notification.repository;

import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.persistence.criteria.*;
import org.chainoptimnotifications.notification.model.Notification;
import org.chainoptimnotifications.notification.model.NotificationUser;
import org.chainoptimnotifications.outsidefeatures.model.PaginatedResults;

import java.util.List;

public class NotificationSearchRepositoryImpl implements NotificationSearchRepository {

    @PersistenceContext
    private EntityManager entityManager;

    @Override
    public PaginatedResults<NotificationUser> findByUserIdAdvanced(String userId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage) {
        CriteriaBuilder builder = entityManager.getCriteriaBuilder();

        // Main Query
        CriteriaQuery<NotificationUser> query = builder.createQuery(NotificationUser.class);
        Root<NotificationUser> notificationUserRoot = query.from(NotificationUser.class);
        Join<NotificationUser, Notification> notificationJoin = notificationUserRoot.join("notification", JoinType.LEFT);
        Predicate conditions = getConditions(builder, notificationUserRoot, notificationJoin, userId, searchQuery);
        query.where(conditions);
        if (sortBy != null && !sortBy.isEmpty()) {
            Path<Object> sortProperty = notificationJoin.get(sortBy);
            Order sortOrder = ascending ? builder.asc(sortProperty) : builder.desc(sortProperty);
            query.orderBy(sortOrder);
        }
        List<NotificationUser> notificationUsers = entityManager.createQuery(query)
                .setFirstResult((page - 1) * itemsPerPage)
                .setMaxResults(itemsPerPage)
                .getResultList();

        // Count Query
        CriteriaQuery<Long> countQuery = builder.createQuery(Long.class);
        Root<NotificationUser> countRoot = countQuery.from(NotificationUser.class);
        countQuery.select(builder.count(countRoot));
        // Reapply joins specifically for count to avoid reuse
        Join<NotificationUser, Notification> countNotificationJoin = countRoot.join("notification", JoinType.LEFT);
        Predicate countConditions = getConditions(builder, countRoot, countNotificationJoin, userId, searchQuery);
        countQuery.where(countConditions);
        long totalCount = entityManager.createQuery(countQuery).getSingleResult();

        return new PaginatedResults<>(notificationUsers, totalCount);
    }


    private Predicate getConditions(CriteriaBuilder builder, Root<NotificationUser> notificationUserRoot, Join<NotificationUser, Notification> notificationJoin, String userId, String searchQuery) {
        Predicate conditions = builder.conjunction();
        if (userId != null) {

            conditions = builder.and(conditions, builder.equal(notificationUserRoot.get("userId"), userId));
        }
        if (searchQuery != null && !searchQuery.isEmpty()) {
            conditions = builder.and(conditions, builder.like(notificationJoin.get("title"), "%" + searchQuery + "%"));
        }
        return conditions;
    }
}