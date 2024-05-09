package org.chainoptimnotifications.core.notification.service;


import org.chainoptimnotifications.core.notification.model.NotificationExtraInfo;
import org.chainoptimnotifications.internal.supply.model.SupplierOrderEvent;

public interface ExtraInfoFormatterService {

    NotificationExtraInfo formatExtraInfo(SupplierOrderEvent orderEvent);
}
