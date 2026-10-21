#pragma once
struct ColumnMicroGroup {int width,members[4];};
struct ColumnMicroState {
 ColumnsArgs columns[24];
 int batch,slots,role,fused,epoch,probe;unsigned long long arrival[4];
 unsigned* ready;unsigned* owner_done;int* remaining;int* abort;
 unsigned long long* marks;unsigned long long* placement;unsigned long long deadline;
 // Candidate-only G05 metadata tail; control and fixed-column paths upload
 // only the common prefix. Unused device tail is never read by those paths.
 ColumnMicroGroup groups[24];int group_for[24];
};
