package org.chainoptimnotifications.core.notification.service;

import org.chainoptimnotifications.internal.in.demand.model.ClientOrderEvent;
import org.chainoptimnotifications.internal.in.supply.model.SupplierOrderEvent;

public interface NotificationListenerService {

    void listenSupplierOrderEvent(SupplierOrderEvent event);
    void listenClientOrderEvent(ClientOrderEvent event);
}
