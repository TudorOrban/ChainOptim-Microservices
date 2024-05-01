package org.chainoptimnotifications.notification.service;

import org.chainoptimnotifications.notification.model.Notification;
import org.chainoptimnotifications.outsidefeatures.model.ClientOrderEvent;
import org.chainoptimnotifications.outsidefeatures.model.SupplierOrderEvent;

public interface NotificationFormatterService {

    Notification formatEvent(SupplierOrderEvent event);
    Notification formatEvent(ClientOrderEvent event);
}
