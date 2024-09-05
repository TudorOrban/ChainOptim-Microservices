package org.chainoptimstorage.core.crate.service;


import org.chainoptimstorage.core.crate.dto.CreateCrateDTO;
import org.chainoptimstorage.core.crate.dto.UpdateCrateDTO;
import org.chainoptimstorage.core.crate.model.Crate;

import java.util.List;

public interface CrateService {

    List<Crate> getCratesByOrganizationId(Integer organizationId);
    Crate createCrate(CreateCrateDTO crateDTO);
    Crate updateCrate(UpdateCrateDTO crateDTO);
    void deleteCrate(Integer crateId);
}
