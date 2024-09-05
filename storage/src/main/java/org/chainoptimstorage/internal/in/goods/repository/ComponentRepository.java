package org.chainoptimstorage.internal.in.goods.repository;


import org.chainoptimstorage.internal.in.goods.model.Component;

import java.util.Optional;

public interface ComponentRepository {

    Optional<Component> findById(Integer id);
}
