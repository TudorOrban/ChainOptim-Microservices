package org.chainoptimnotifications.core.notification.service;

import org.chainoptimnotifications.core.notification.model.Notification;
import org.chainoptimnotifications.internal.demand.model.ClientOrderEvent;
import org.chainoptimnotifications.internal.supply.model.SupplierOrderEvent;

public interface NotificationFormatterService {

    Notification formatEvent(SupplierOrderEvent event);
    Notification formatEvent(ClientOrderEvent event);
}
