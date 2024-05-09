package org.chainoptimnotifications.core.notification.service;

import org.chainoptimnotifications.core.notification.model.NotificationUserDistribution;
import org.chainoptimnotifications.internal.in.demand.model.ClientOrderEvent;
import org.chainoptimnotifications.internal.in.supply.model.SupplierOrderEvent;

public interface NotificationDistributionService {

    NotificationUserDistribution distributeEventToUsers(SupplierOrderEvent event);
    NotificationUserDistribution distributeEventToUsers(ClientOrderEvent event);
}
