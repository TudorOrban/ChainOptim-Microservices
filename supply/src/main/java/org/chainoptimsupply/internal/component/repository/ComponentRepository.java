package org.chainoptimsupply.internal.component.repository;


import org.chainoptimsupply.internal.subscriptionplan.model.Component;

import java.util.Optional;

public interface ComponentRepository {

    Optional<Component> findById(Integer id);
}
