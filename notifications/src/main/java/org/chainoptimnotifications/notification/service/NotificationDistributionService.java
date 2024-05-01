package org.chainoptimnotifications.notification.service;

import org.chainoptimnotifications.notification.model.NotificationUserDistribution;
import org.chainoptimnotifications.outsidefeatures.model.ClientOrderEvent;
import org.chainoptimnotifications.outsidefeatures.model.SupplierOrderEvent;

public interface NotificationDistributionService {

    NotificationUserDistribution distributeEventToUsers(SupplierOrderEvent event);
    NotificationUserDistribution distributeEventToUsers(ClientOrderEvent event);
}
