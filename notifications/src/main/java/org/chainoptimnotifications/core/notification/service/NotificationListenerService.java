package org.chainoptimnotifications.core.notification.service;

import org.chainoptimnotifications.internal.demand.model.ClientOrderEvent;
import org.chainoptimnotifications.internal.supply.model.SupplierOrderEvent;

public interface NotificationListenerService {

    void listenSupplierOrderEvent(SupplierOrderEvent event);
    void listenClientOrderEvent(ClientOrderEvent event);
}
