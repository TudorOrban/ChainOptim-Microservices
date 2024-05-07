package org.chainoptimsupply.internal.subscriptionplan.service;


import org.chainoptimsupply.kafka.Feature;

public interface SubscriptionPlanLimiterService {

    boolean isLimitReached(Integer organizationId, Feature feature, Integer quantity);
}
