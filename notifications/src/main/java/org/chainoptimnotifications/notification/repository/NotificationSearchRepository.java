package org.chainoptimnotifications.notification.repository;

import org.chainoptimnotifications.notification.model.NotificationUser;
import org.chainoptimnotifications.outsidefeatures.model.PaginatedResults;

public interface NotificationSearchRepository {

    PaginatedResults<NotificationUser> findByUserIdAdvanced(String userId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage);
}
