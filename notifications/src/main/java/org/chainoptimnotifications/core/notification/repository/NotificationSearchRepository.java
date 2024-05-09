package org.chainoptimnotifications.core.notification.repository;

import org.chainoptimnotifications.core.notification.model.NotificationUser;
import org.chainoptimnotifications.shared.search.model.PaginatedResults;

public interface NotificationSearchRepository {

    PaginatedResults<NotificationUser> findByUserIdAdvanced(String userId, String searchQuery, String sortBy, boolean ascending, int page, int itemsPerPage);
}
