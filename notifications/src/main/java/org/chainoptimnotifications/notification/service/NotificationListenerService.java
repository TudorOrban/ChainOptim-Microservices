package org.chainoptimnotifications.notification.service;

import org.chainoptimnotifications.outsidefeatures.model.ClientOrderEvent;
import org.chainoptimnotifications.outsidefeatures.model.SupplierOrderEvent;

public interface NotificationListenerService {

    void listenSupplierOrderEvent(SupplierOrderEvent event);
    void listenClientOrderEvent(ClientOrderEvent event);
}
