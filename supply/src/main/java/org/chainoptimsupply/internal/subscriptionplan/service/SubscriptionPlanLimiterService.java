package org.chainoptimsupply.internal.subscriptionplan.service;


import org.chainoptimsupply.shared.enums.Feature;

public interface SubscriptionPlanLimiterService {

    boolean isLimitReached(Integer organizationId, Feature feature, Integer quantity);
}
