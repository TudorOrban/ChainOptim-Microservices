package org.chainoptimnotifications.notification.service;


import org.chainoptimnotifications.notification.model.NotificationExtraInfo;
import org.chainoptimnotifications.outsidefeatures.model.SupplierOrderEvent;

public interface ExtraInfoFormatterService {

    NotificationExtraInfo formatExtraInfo(SupplierOrderEvent orderEvent);
}
