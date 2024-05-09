package org.chainoptimsupply.internal.in.goods.repository;


import org.chainoptimsupply.internal.in.goods.model.Component;

import java.util.Optional;

public interface ComponentRepository {

    Optional<Component> findById(Integer id);
}
