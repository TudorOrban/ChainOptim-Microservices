package org.chainoptimstorage.core.compartment.model;

import org.chainoptimstorage.core.crate.model.CrateData;
import org.chainoptimstorage.core.crate.model.CrateSpec;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CompartmentData {

    private List<CrateSpec> crateSpecs;
    private List<CrateData> currentCrates;
}
