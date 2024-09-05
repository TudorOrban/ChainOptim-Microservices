package org.chainoptimstorage.internal.in.tenant.service;

import org.chainoptimstorage.shared.kafka.OrganizationEvent;

public interface OrganizationChangesListenerService {

    void listenToOrganizationEvent(OrganizationEvent event);
}
