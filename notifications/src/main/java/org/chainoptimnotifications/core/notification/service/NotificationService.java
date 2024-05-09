package org.chainoptimnotifications.core.notification.service;


import org.chainoptimnotifications.internal.demand.model.ClientOrderEvent;
import org.chainoptimnotifications.internal.supply.model.SupplierOrderEvent;

public interface NotificationService {

    void createNotification(SupplierOrderEvent event);
    void createNotification(ClientOrderEvent event);
}
