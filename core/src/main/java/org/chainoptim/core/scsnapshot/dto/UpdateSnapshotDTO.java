package org.chainoptim.core.scsnapshot.dto;

import org.chainoptim.core.scsnapshot.model.Snapshot;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UpdateSnapshotDTO {

    private Integer id;
    private Integer organizationId;
    private Snapshot snapshot;
}
