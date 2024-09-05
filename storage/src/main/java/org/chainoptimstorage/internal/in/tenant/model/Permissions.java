package org.chainoptimstorage.internal.in.tenant.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serial;
import java.io.Serializable;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Permissions implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private FeaturePermissions organization;
    private FeaturePermissions products;
    private FeaturePermissions factories;
    private FeaturePermissions warehouses;
    private FeaturePermissions suppliers;
    private FeaturePermissions clients;
}
